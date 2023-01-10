(*DICIEMBRE - 2010*)

let x = [1;2;3] in 4::x;;
(*- : int list = [4; 1; 2; 3]*)

List.hd x;;
(*error variable indefinida x*)

let x = let x = [1;2;3] in List.tl x;;
(*val x : int list = [2; 3]*)

let y::x = x;;
(*val y: int = 2
  val x: int list = [3]*)

let (y,x) = (x,y::[]) in y@x;;
(*- : int list = [3; 2]*)

let x y = y::[] in x y;;
(*- : int list = [2]*)

(function f -> f (f 2.0)) (function f -> f ** f);;
(*- : float = 256. *)

let rec f n = if n > 1 then n * f (n-2) else n;;
(*val f : int -> int = <fun>*)

f 4 + f 5;;
(*- : int = 15*)

let rec f = function n -> n * f (n - 2) | 1 -> 1 | 0 -> 0;;
(*val f : int -> int = <fun> *)

4 + f 5;;
(*BUCLE INFINITO*)

let f = function 0 -> 0 | 1 -> 1 | n -> n * f (n - 2);;
(*val f : int -> int = <fun>*)

f 4 + f 5;;
(*BUCLE INFINITO*)

let rec ap op = function [] -> [] | cab::col -> op cab::ap op col;;
(*val : ('a list -> 'b list) -> 'a list -> 'b list = <fun>*)

let pri (a,_) = a;;
(*val pri : ('a * 'b) -> 'a = <fun>*)

ap pri;;
(*- : ('a * 'b) list -> 'a list = <fun>*)

ap pri [([],([],[]))];;
(*- : 'a list = [[]]*)


(*Defina las funciones ldoble, lcuadrado  y  lcubo  de forma que aplicadas a una lista de flotantes den,
repectivamente, la lista de sus dobles, de sus cuadrados y de sus cubos.Y defina las funciones lsuma 
y  lproducto de forma que aplicadas a una lista de flotantes den, respectivamente, la suma y el producto
de todos sus elementos.*)

let rec ldoble = function 
    [] -> []
  | h::t -> (2. *. h):: (ldoble t);;

let ldoble' l = 
  let rec aux l1 sol = match l1 with
      [] -> List.rev sol
    | h::t -> aux t ((2. *. h)::sol)
  in aux l [];; 

let rec lcuadrado = function 
    [] -> []
  | h::t -> (h**2.)::(lcuadrado t);;

let rec lcubo = function 
    [] -> []
  | h::t -> (h**3.)::(lcubo t);;

let rec lsuma = function 
    [] -> 0.
  | h::t -> h +. (lsuma t);;

let lsuma l = 
  let rec aux l1 acum = match l1 with
      [] -> acum
    | h::t -> aux t (acum +. h)
  in aux l 0.;;

let lproducto l = 
  let rec aux acum = function 
      [] -> acum
    | h::t -> aux (h *. acum) t 
  in aux 1. l;; 

(*a. Indique el tipo de las funciones  filtro  y  no_pertenece_a, definidas a continuación.
  b. Utilizando las funciones  filtro y  no_pertenece_a,escriba una definición lo más breve 
  posible de la función  diferencia: ‘a list->‘a list-> ‘a list, de forma que diferencia l1 l2 
  sea la lista de los elementos de l1 que no están en l2.*)

let rec filtro f = function 
    [] -> [] 
  | h::t -> if f h then h::filtro f t 
            else filtro f t;;
(*val filtro : ('a -> bool) -> 'a list -> 'a list = <fun>*)

let rec no_pertenece_a = function 
    [] -> (function _ -> true)
  | h::t -> (function x -> x <> h && no_pertenece_a t x);;
(*val no_pertenece_a : 'a list -> 'a -> bool = <fun> *)

let rec diferencia l1 l2 = match l1 with
    [] -> []
  | h::t -> if no_pertenece_a l2 h then h::diferencia t l2 
            else diferencia t l2;;

let diferencia' l1 l2 = filtro (no_pertenece_a l2) l1;;



(*SEPTIEMBRE - 2010*)

let apa x f = f x;;
(*val apa : 'a -> ('a -> b') -> 'b = <fun>*)

List.map (apa 2) [(function x -> x * x); succ; (+)1; (-) 1];; 
(*- : int list = [4; 3; 3; 1]*)

let apa_rep n x f = 
  let rec aux x = 
    function 0 -> x
    | n -> aux (apa x f) (n-1)
  in aux x (abs n);;
(*val apa_rep : int -> 'a -> ('a -> 'a) -> 'a = <fun>*)

apa_rep (-2) "x" (function x -> x ^ x);;
(*- : String = "xxx"*)
(*aux "x" 2 = aux "xx" 1 = aux "xxxx" 0 = "xxxx"*)

let fop op f g = function y -> op (f y) (g y);;
(*val fop : ('a -> 'b -> 'c) -> ('d -> 'a) -> ('d -> 'b) -> 'd -> 'c = <fun>*)

let suma = fop (+);;
(*val suma : ('a -> int) -> ('a -> int) -> 'a -> int = <fun>*)

let f = let f1 x = x * x in
        let f2 x = f1 x * x in
        suma f1 f2
in f 2;;
(*- : int = 12*)

(*Redefina la función f de modo que sólo se utilice recursividad terminal 
(pero debe dar siempre el mismo resultado que la original)*)

let rec f orden = function 
    [] -> raise (Failure "f") 
  | [x] -> x 
  | h::t -> let m = f orden t in 
                    if orden h m then h else m;;

let f' orden = function 
    [] -> raise (Failure "f")
  | h::t -> let rec aux l sol = match l with 
            [] -> sol
          | [x] -> if orden x sol then x else sol
          | h1::t1 -> if orden h1 sol then aux t1 h1 else aux t1 sol
          in aux (h::t) h;;

(*Una relación (de equivalencia, de orden, etc...) en un conjunto  A puede representarse 
   como una función de (A × A) → bool , que indica para cada pareja de elementos de A si 
   están o no relacionados. Dada una función cualquiera f : A → B, puede hablarse de la 
   relación de equivalencia que induce sobre elconjunto A como aquella en la que son 
   equivalentes los elementos que tienen la misma imagen.*)

(*a. Defina en ocaml una función  rel_eq : ('a -> 'b) -> 'a * 'a -> bool ,  que para 
   cualquier función devuelva la relación de equivalencia inducida por ella en el sentido 
   señalado.*)

let rel_eq f (a,b) = (f a = f b);; 

(*b. Defina en ocaml una función clases_eq : ('a * 'a -> bool) -> 'a list -> 'a list list , 
   de modo que, dada una relación de equivalencia r sobre un conjunto (tipo de dato) A y dada 
   una lista de elementos de A , "divida" los elementos de la lista en clases de equivalencia 
   inducidas por la relación r.(El orden no sería relevante, aunque sí el número de apariciones 
   de cada elemento; así, por ejemplo, 
   clases_eq (function (x,y) -­> x mod 2 = y mod 2) 
   [5;7;9;10;30;0;4;1;5;10] 
   podría ser la lista [[10; 30; 0; 4; 10]; [5; 7; 9; 1; 5]]).*)

let rec clases_eq r =
  let rec anadir x = function
    [] -> [[x]]
    | (h::t)::clases -> if r x h then (x::h::t)::clases
                      else (h::t)::(anadir x clases)
    in function
      [] -> []
      | h::t -> anadir h (clases_eq r t);;



(*ENERO - 2014*)

(*Dada la siguiente defincion de tipo de dato 'a arbol de árbol binario:*)
(*a) defina una funcion cont: 'a->'a arbol -> int que devuelva el numero
   de nodos de un arbol que están etiquetados con un valor determinado.*)

   type 'a arbol = Vacio | Nodo of ('a * 'a arbol * 'a arbol);;

   let rec cont x = function
       Vacio -> 0
     | Nodo (a,l,r) -> 
       let left, right = cont x l,cont x r in 
       if a = x then (1 + left + right)
       else (left + right);;
   
   (*b) defina una funcion subst: 'a -> 'a -> 'a arbol -> 'a arbol de forma que 
      aplicada a un arbol devuelva un arbol igual al orginal salvo los nodos 
      que tuviesen valor de x tendrán valor de y*)
   
   let rec subst x y = function
       Vacio -> Vacio
     | Nodo (a,r,l) -> 
       let left, right = subst x y l, subst x y r in
       if a = x then Nodo (y,left, right)
       else Nodo (a, left, right);;
   
   (*Defina una funcion l_ordenada: ('a -> 'a -> bool) -> a' list -> bool
      de forma que si f es una relacion de orden en el tipo 'a, l_ordenada f
      sea la funcion que dice si una lista está ordenada según el orden f*)
   
   let rec l_ordenada f = function 
       [] -> true
     | [h] -> true
     | h1::h2::t -> f h1 h2 && l_ordenada f (h2::t);;
     
   (*Defina utilizado exclusivamente recursividad terminal una funcion 
      l_max: 'a list -> 'a que devuelva de cada lista el mayor de sus
      elementos.*)
   
   let rec l_max = function
       [] -> raise (Failure "max")
     | h::t -> let rec aux m = function 
               [] -> m 
               | h::t -> if h > m then aux h t
                         else aux m t 
               in aux h t;;



(*JUNIO - 2014*)

(*Funcion suma que suma los elementos de dos listas respectivamente.
   Ej. suma [2;5;4;9] [2;4;3] es igual a [4;9;7;9];;*)

   let suma l1 l2 = 
    let rec aux l1 l2 lsum = match l1, l2 with
      [], [] -> List.rev lsum
    | [], h::t -> aux l1 t (h::lsum)
    | h::t, [] -> aux t l2 (h::lsum)
    | h1::t1, h2::t2 -> aux t1 t2 ((h1+h2)::lsum)
  in aux l1 l2 [];;
  
  (*Dados estes dos tipos de datos definir dos funciones, para pasar
     de un tipo a otro y viceversa*)
  
  type 'a a2 = AO of 'a 
          | AIz of 'a * 'a a2
          | ADc of 'a * 'a a2
          | A2 of 'a * 'a a2 * 'a a2;;
  
  type 'a abin = V | N of 'a * 'a abin * 'a abin;;
  
  let rec a2_of_abin = function
      V -> raise (Failure "a2_of_abin")
    | N(r,V,V) -> AO r
    | N(r,i,V) -> AIz (r,a2_of_abin i)
    | N(r,V,d) -> ADc (r,a2_of_abin d)
    | N(r,i,d) -> A2 (r, a2_of_abin i, a2_of_abin d);;
  
  let rec abin_of_a2 = function 
      AO r -> N (r,V,V)
    | AIz(r,i) -> N (r,abin_of_a2 i,V)
    | ADc(r,d) -> N (r,V,abin_of_a2 d)
    | A2(r,i,d) -> N (r, abin_of_a2 i, abin_of_a2 d);;



(*ENERO - 2022*)

let f3 f x = (x, f x, f (f x));;
(*val f3 : ('a -> 'a) -> 'a -> 'a * 'a * 'a = <fun>*)

let x, y, z = let g x = x * x in f3 g 2;;
(*val x : int = 2
val y : int = 4
val z : int = 16
*)

(function _ :: _ :: t -> t) [1; 2; 3];;
(*- : int list = [3]*)

List.map (function x -> 2 * x + 1);;
(*- : int list -> int list = <fun>*)

let rec f = function [] -> 0 | h::[] -> h
  | h1::h2::t -> h1 + h2 - f t;;
(*val f = int list -> int = <fun>*)

f [1000; 100; 10], f [1000; 100; 10; 1];;
(*- : (int * int) = (1090, 1089)*)

List.fold_right (-) [4; 3; 2] 1;;
(*- : int = 2*)



let rec comb f = function 
    h1::h2::t -> f h1 h2 :: comb f t 
  | l -> l;;
(*val comb : ('a -> 'a -> a) -> a'list -> 'a list = <fun>*)

comb (+);;
(*- : int list -> int list = <fun>*)

comb (+) [1; 2; 3; 4; 5];;
(*- : int list = [3; 7; 5]*)


(*Defina la funcion comb anterior de forma recursiva terminal*)

let comb' f l = 
  let rec aux l1 sol = match l1 with
    [] -> List.rev sol
  | [h] -> List.rev (h::sol)
  |  h1::h2::t -> aux t ((f h1 h2)::sol)
  in aux l [];;


(*Partiendo de la definicion:*)

type 'a tree = T of 'a * 'a tree list;;

let s x = T (x,[]);;
(*val s : 'a -> 'a tree = <fun> *)

let t = T (1, [s 2; s 3; s 4]);;
(*val t : int tree = T (1, [T (2, []); T (3, []); T (4, []))*)

let rec sum = function 
    T (x,[]) -> x
  | T (r, T(r1, l)::t) -> r + sum (T (r1, l @ t));;
(*val sum : int tree -> int = <fun>*)

sum t;;
(*- : int = 10*)

(*Defina la funcion sum anterior de forma recursiva terminal*)

let sum' t = 
  let rec aux t s = match t with
    T(x,[]) -> x + s
  | T(x, T(y,l)::t) -> aux (T(y, List.rev_append (List.rev l)t)) (x + s)
  in aux t 0;;
