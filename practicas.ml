
PRÁCTICA 1 - expr1.ml, expr2.ml, nombre.ml

();;
(*- : unit = ()*)

2 + 5 * 3;;
(*- : int = 17*)

1.0;;
(*- : float = 1*)

(*1.0 * 2;;*)
(*Error de tipos: Se está utilizando un operador de enteros para
  multiplicar un float a la izquierda por un entero.*)

(*2 - 2.0;;*)
(*Error de tipos: Se está utilizando un operador de enteros para 
   estar a la derecha del operador un float.*)

(*3.0 + 2.0;;*)
(*Error de tipos: Se está utilizando un operador de enteros para 
  sumar dos float*)

5 / 3;;
(*- : int = 1*)

5 mod 3;;
(*- : int = 2*)

3.0 *. 2.0 ** 3.0;;
(*- : float = 24*)

3.0 = float_of_int 3;;
(*- : bool = true *)

(*sqrt 4;;*)
(*Error de tipos: La función sqrt va de float a float y se está 
 intentando aplicar a un entero*)

int_of_float 2.1 + int_of_float (-2.9);;
(*- : int = 0*)

truncate 2.1 + truncate (-2.9);;
(*- : int = 0*)

floor 2.1 +. floor (-2.9);;
(*- : float = -1*)

(*ceil 2.1 +. ceil -2.9;;*)
(*Error sintáctico: Falta un paréntesis en el segundo ceil para
  hacerlo parte del número, sino se interpreta como una resta*)

2.0 ** 3.0 ** 2.0;;
(*- : float = 512*)

'B';;
(*- : char = 'B'*)

int_of_char 'A';;
(*- : int = 65*)

char_of_int 66;;
(*- : char = 'B'*)

Char.code 'B';;
(*- : int = 66*)

Char.chr 67;;
(*- : char = 'C'*)

'\067';;
(*- : char = 'C'*)

Char.chr (Char.code 'a' - Char.code 'A' + Char.code 'M');;
(*- : char = 'm'*)

"this is a string";;
(*- : string = "this is a string"*)

String.length "longitud";;
(*- : int = 8*)

(*"1999" + "1";;*)
(*Error de tipos: Se está utilizando un operador de enteros para 
intentar sumar dos strings*)

"1999" ^ "1";;
(*- : string = "19991"*)

int_of_string "1999" + 1;;
(*- : int = 2000*)

"\064\065";;
(*- : string = "@A"*)

string_of_int 010;;
(*- : string = "10"*)

not true;;
(*- : bool = false*)

true && false;;
(*- : bool = false*)

true || false;;
(*- : bool = true*)

(1 < 2) = false;;
(*- : bool = false*)

"1" < "2";;
(*- : bool = true*)

2 < 12;;
(*- : bool =  true*)

"2" < "12";; 
(*- : bool = false*)

"uno" < "dos";;
(*- : bool = false*)

if 3 = 4 then 0 else 4;;
(*- : int = 4*)

if 3 = 4 then "0" else "4";;
(*- : string = "4"*)

(*if 3 = 4 then 0 else "4";;*)
(*Error de tipos: Debido al 0 después del then, se espera que
   la resolución del if sea un int, sin embargo en el caso del 
   else se devolvería un string.*)

(if 3 < 5 then 8 else 10) + 4;;
(*- : int = 12*)

2.0 *. asin 1.0;;
(*- : float = 3.14159265358979312*)

sin (2.0 *. asin 1.0 /. 2.);; 
(*- : float 1.*)
(*Equivalente a: sin ((2.0 *. asin 1.0) /. 2.);;*)

function x -> 2 * x;;
(*- : int -> int = <fun>*)

(function x -> 2 * x) (2 + 1);;
(*- : int = 6*)

let x = 1;;
(*val x : int = 1*)

let y = 2;;
(*val y : int 2*)

x - y;;
(*- : int = -1*)

let x = y in x - y;;
(*- : int = 0*)

x - y;;
(*- : int = -1*)

(*z;;*)
(*Error de inicialización: El valor z no fue inicializado.*)

let z = x + y;;
(*val z : int = 3*)

z;;
(*- : int = 3*)

let x = 5;;
(*val x : int = 5*)

z;;
(*- : int = 3*)

let y = 5 in x + y;;
(*- : int = 10*)

x + y;;
(*- : int = 7*)

let x = x + y in let y = x * y in x + y + z;;
(*- : int = 24*)

x + y + z;;
(*- : int 10*)

int_of_float;;
(*- : float -> int = <fun>*)

float_of_int;;
(*- : int -> float = <fun>*)

int_of_char;;
(*- : char -> int = <fun>*)

char_of_int;;
(*- : int -> char = <fun>*)

abs;;
(*- : int -> int = <fun>*)

sqrt;;
(*- : float -> float = <fun>*)

truncate;;
(*- : float -> int = <fun>*)

ceil;;
(*- : float -> float = <fun>*)

floor;;
(*- : float -> float = <fun>*)

Char.code;;
(*- : char -> int = <fun>*)

Char.chr;;
(*- : int -> char = <fun>*)

int_of_string;;
(*- : string -> int = <fun>*)

string_of_int;;
(*- : int -> string = <fun>*)

String.length;;
(*- : string -> int = <fun>*)

let f = function x -> 2 * x;;
(*val f : int -> int = <fun>*)

f (2+1);;
(*- : int = 6*)

f 2 + 1;;
(*- : int 5*)

let n = 1;;
(*val n : int = 1*)

let g x = x + n;;
(*val g : int -> int = <fun>*)

g 3;;
(*- : int = 4*)

let n = 5;;
(*val n : int = 5*)

g 3;;
(*- : int = 4*)

let l = function r -> let pi = 2.0 *. asin 1.0 in 2.0 *. pi *. r;;
(*val l : float -> float = <fun>*)

l 3.0;;
(*- : float = 18.8495559215387587*)

(*l 2;;*)
(*Error de tipos: La función l va de float en float y se le está
  introduciendo como argumento un int*)

(*pi;;*)
(*Error de inicialización: Se le fue asignado a pi un valor temporal
   dentro de la función l con in. Fuera de l pi no está inicializado.*)

let pi = 2.0 *. asin 1.0;;
(*val pi : float = 3.14159265358979312*)

pi;;
(*- : float = 3.14159265358979312*)

let v = function r -> pi *. r ** 2.0;;
(*val v : float -> float = <fun>*)

v 2.0;;
(*- : float = 12.5663706143591725*)

--------------------------------------------------------------

let u = 3 + 2 - 5 * 4 / 2;;
(*val u : int = -5*)

let v = asin (-1.);;
(*val v : float = -1.57079632679489656*)

let w = char_of_int (67 - 2);;
(*val w = char = 'A'*)

let x = (int_of_char('3') < int_of_char('6') + int_of_char('1'));;
(*val x : bool = true*)

let y = if (3 < 2) then "3 menor que 2" else "3 mayor que 2";;
(*val y : string "3 mayor que 2"*)

----------------------------------------------------------------

let u = 3 + 2 - 5 * 4 / 2;;
(*val u : int = -5*)

let v = asin (-1.);;
(*val v : float = -1.57079632679489656*)

let w = char_of_int (67 - 2);;
(*val w = char = 'A'*)

let x = (int_of_char('3') < int_of_char('6') + int_of_char('1'));;
(*val x : bool = true*)

let y = if (3 < 2) then "3 menor que 2" else "3 mayor que 2";;
(*val y : string "3 mayor que 2"*)



PRÁCTICA 2 - e.ml, fact.ml, pi.ml

let e = (1.+.1./.10000000.)**(10000000.)
in print_endline (string_of_float e)

let rec fact = function
0 -> 1
| n -> n * fact (n - 1) 
in 
if Array.length Sys.argv = 2
  then print_endline (string_of_int (fact (int_of_string(Sys.argv.(1)))))
else
  print_endline ("Error: número de argumentos inválido.")

let pi = 2. *. asin 1.0 in 
  print_endline (string_of_float pi)



PRÁCTICA 3 - condis.ml, ej31.ml, ej33.ml, fib.ml, prime.ml

false && (2 / 0 > 0);;
(*- : bool = false*)

true && (2 / 0 > 0);;
(*Error de ejecución: no es posible dividir entre 0*)

true || (2 / 0 < 0);;
(*- : bool = true*)

false || (2 / 0 > 0);;
(*Error de ejecución: no es posible dividir entre 0*)

let con b1 b2 = b1 && b2;;
(*val con : bool -> bool -> bool = <fun>*)

let dis b1 b2 = b1 || b2;;
(*val dis : bool -> bool -> bool = <fun>*)

con (1 < 0) (2 / 0 > 0);;
(*Error de ejecución: no es posible dividir entre 0. 
   Evalúa ambas expresiones antes de pasarlas a la función.*)

(1 < 0) && (2 / 0 > 0);;
(*- : bool = false*)

dis (1 > 0) (2 / 0 > 0);;
(*Error de ejecución: no es posible dividir entre 0. 
   Evalúa ambas expresiones antes de pasarlas a la función.*)

(1 > 0) || (2 / 0 > 0);;
(*- : bool true*)

---------------------------------------------------------------------

(*let f1 n = if n mod 2 = 0 then n / 2 else 2 * n;;

let f2 n = if n mod 2 = 0 then "es par" else "es impar";;

let f3 n = if n mod 2 = 0 then "m ́ultiplo de 2"
          else if n mod 3 = 0 then "m ́ultiplo de 3"
          else "impar";;*)

let f1 n = (function true -> n/2 | false -> 2*n) (n mod 2 = 0);;

let f2 n = (function true -> "es par" | false -> "es impar") (n mod 2 = 0);;

let f3 n = (function true -> "multiplo de 2" | false -> (function true -> "multiplo de 3" 
| false -> "impar")(n mod 3 = 0))(n mod 2 = 0);;

-------------------------------------------------------------------------------------

(*let g n = (n >= 0 && n mod 2 = 0) || n mod 2 = -1;;*)

let g1 n = if (if n >= 0 then n mod 2 = 0 else false) then true else n mod 2 = -1;;

let g2 n = (function true -> true | false -> n mod 2 = -1) ((function true -> n mod 2 = 0 
| false -> false) (n >= 0));;

----------------------------------------------------------------------------------------

let rec fib n =
  if n <= 1 then n
  else fib (n-1) + fib (n-2);;


let rec print_fib n =
  if n = 0 then ()
  else print_fib (n-1);
  print_endline (string_of_int(fib (n)));;
  if Array.length Sys.argv = 2
    then print_fib (int_of_string(Sys.argv.(1)))
  else
    print_endline ("Error.");;
    
-------------------------------------------------------------------------------------------

let is_prime n =
  let rec check_from i =
  i >= n ||
  (n mod i <> 0 && check_from (i+1))
  in check_from 2;;

let rec next_prime n = 
  let n = n + 1  in
  if is_prime(n) then n
  else next_prime (n);;

let rec last_prime_to n =
  if is_prime (n) then n
  else let n = n -1 in
  last_prime_to (n);;

let rec is_prime2 n = 
let raiz = int_of_float(sqrt(float_of_int(n))) in
let rec check_from i =
  if i > raiz then true
  else (n mod i <> 0 && check_from (i+1))
  in check_from 2;;



PRÁCTICA 4 - ej41.ml, mcd.ml, power.ml

let rec sum_cifras n =
  if n / 10 = 0 then n
  else sum_cifras(n / 10) + n mod 10;; 
  
let rec num_cifras n = 
if n / 10 = 0 then 1
else num_cifras(n / 10) + 1;;

let rec exp10 n = 
  if n = 0 then 1
  else 10 * exp10 (n - 1);;

let rec reverse n = 
  if num_cifras n = 1 then n
  else n mod 10 * exp10 (num_cifras(n)-1) + reverse (n / 10);;
  
let rec palindromo s = 
  if String.length s = 0 || String.length s = 1 then true
  else s.[0] = s.[String.length (s) - 1] && 
  palindromo (String.sub s 1 ((String.length (s) - 1) - 1));;

---------------------------

let rec mcd (x,y) = 
  if y = 0 then x
  else mcd (y, (x mod y));;

------------------------------

let rec power x y = 
  if y = 0 || x = 1 then 1
  else power x (y-1) * x;;

let rec power' x y =
  if y = 0 || x = 1 then 1
  else if y mod 2 = 0 then
    power' (x * x) (y / 2)
  else power' (x * x) (y / 2) * x;;

(*power' es mejor que power en cuanto a términos de eficiencia porque se hacen menos 
    llamadas recursivas a la función power o power'. Mientras que en power se llama
    a la función y veces, en power' se llama log en base 2 de y veces.
    Por otra parte, al estar trabajando con int y realizar potencias, llegaremos muy
    rápido a maxint, por lo que realmente no merece la pena una implementación más 
    rápida.*)

let rec powerf x n =
  if n = 0 || x = 1. then 1.
  else if n mod 2 = 0 then
    powerf (x *. x) (n / 2)
  else (powerf (x *. x) (n / 2)) *. x;;



PRÁCTICA 5 - collatz.ml 

let f n = if n mod 2 = 0 then n / 2 else 3 * n + 1;;

let rec orbit n = 
  if n = 1 then string_of_int 1
  else string_of_int n ^ ", " ^ orbit (f n);;

let rec length n = 
  if n = 1 then 0
  else length (f n) + 1;;

let rec top n =
  if n = 1 then n
  else max n (top(f n));;

let rec length'n'top n =
  if n = 1 then (0,n)
  else 
    let anterior = length'n'top (f n)
  in ((fst anterior) + 1, max n (snd anterior));;

let rec longest_in m n = 
  let lm = length m in
  if m = n then (m, lm)
  else let i,li = longest_in (m + 1) n in
  if li > lm then (i,li)
  else (m, lm);;

let rec highest_in m n = 
  let tm = top m in
  if m = n then (n, tm)
  else let i,ti = highest_in (m + 1) n in
  if ti > tm then (i, ti)
  else (m, tm);;



PRÁCTICA 6 - ej62.ml, powmod.ml

let curry f a b = f (a, b);;

let uncurry f (a,b) = f a b;;

uncurry (+);;
(*- : int * int -> int = <fun>*)

let sum = (uncurry (+));;
(*val sum : int * int -> int = <fun>*)

(*sum 1;;*)
(*Error de tipos: la función sum espera un valor int * int pero
   recibe un int*)

sum (2,1);;
(*- : int = 3*)

let g = curry (function p -> 2 * fst p + 3 * snd p);;
(*val g : int -> int -> int = <fun>*)

(*g (2,5);;*)
(*Error de tipos: es una función curry por lo que, en
este caso, espera dos enteros, no un par de enteros*)

let h = g 2;;
(*val h : int -> int = <fun>*)

h 1, h 2, h 3;;
(*- : int * int * int = (7, 10, 13)*)
(*Creamos un triplete con los tres resultados de aplicar h a 
   distintos enteros*)

let comp f g h = f (g (h));;

let f = let square x = x * x in comp square ((+) 1);;
(*val f : int -> int = <fun>*)
(*Calcula el cuadrado de la suma del argumento + 1*)

f 1, f 2, f 3;;
(*- : int * int * int = (4, 9, 16)*)

let i a = a;;

let j (a, b) = a;;

let k (a, b) = b;;

let l a = a::[];;

(*Hay una funcion sola para cada tipo excepto para el último donde habría tantas funciones
   como elementos de una lista posibles en ocaml.*)

-------------------------------------------------------------------------------------------

let rec power' b e =
  if e = 0 || b = 1 then 1
  else if e mod 2 = 0 then
    power' (b * b) (e / 2) 
  else power' (b * b) (e / 2) * b;;
  
let rec powmod m b e =
  if e = 0 then 1 mod m (*en caso 1 mod 1 debe dar 0, no 1*)
  else if e mod 2 = 0 then
    powmod m (b * b) (e / 2) mod m
  else powmod m (b * b) (e / 2) * b mod m;;



PRÁCTICA 7 - mylist.ml (NO TERMINAL)

let hd = function
  [] -> raise (Failure "hd")
  | h::t -> h;;

let tl = function 
  [] -> raise (Failure "tl")
  | h::t -> t;;

let rec length = function 
  [] -> 0
  | h::t -> 1 + length t;;

let rec compare_lengths l1 l2 = match l1, l2 with
  [], [] -> 0
  | h::t, [] -> 1
  | [], h::t -> -1
  | h1::t1, h2::t2 -> compare_lengths t1 t2;;

let rec nth l p = match l with
h::t -> if p = 0 then h 
        else if p < 0 then raise (Invalid_argument "nth")
        else nth t (p - 1)
|[]-> raise (Failure "nth");;

let rec append l1 l2 = match l1 with
  [] -> l2
  | h::t -> h :: append t l2;;

let rec find f l = match l with
  [] -> raise (Failure "Not_found")
  | h::t -> if f h then h else find f t;;

let rec for_all f l = match l with
  [] -> true
  | h::t -> f h && for_all f t;;

let rec exists f l = match l with
  [] -> false
  | h::t -> f h || exists f t;;

(*Funcion para prueba find, for_all, exists
let f x = 
  if x > 0 then true
  else false;*)

let rec mem a set = match set with
  [] -> false
  | h::t -> a = h || mem a t;;

let rec filter f l = match l with
  [] -> []
  | h::t -> if f h then h::filter f t
            else filter f t;;

let rec find_all f l = filter f l;;

let rec partition f l = match l with
  [] -> [],[]
  | h::t -> let p1,p2 = partition f t in 
            if f h then h::p1,p2
            else p1,h::p2;;
  
let rec split l = match l with 
  [] -> [],[]
  | h::t -> let p1,p2 = split t in 
            fst h::p1,snd h::p2;;

let rec combine l1 l2 = match l1, l2 with 
  [],[] -> []
  | h::t,[] -> raise (Invalid_argument "combine")
  | [],h::t -> raise (Invalid_argument "combine")
  | h1::t1,h2::t2 -> (h1,h2)::combine t1 t2;;

let rec init n f = 
  if n < 0 then raise (Invalid_argument "init")
  else if n = 0 then []
  else f 0 :: init(n-1)(function i -> f (i+1));;

let rev l = 
  let rec aux l1 l2 = match l1 with
    [] -> l2
    | h::t -> aux t (h::l2) in
  aux l [];;

let rec rev_append l1 l2 = match l1 with 
  [] -> l2
  | h::t -> rev_append t (h::l2);;

let rec concat l = match l with
  [] -> []
  | (h1::t1)::t -> h1::concat(t1::t)
  | []::t -> concat t;;

let flatten l = concat l;;

let rec map f l = match l with
  [] -> []
  | h::t -> f h :: map f t;;

(*funcion para prueba de map
let f = function x -> 2*x;;*)

let rev_map f l = 
  let rec aux l1 l2 = match l1 with 
    [] -> l2
    | h::t -> aux t ((f h)::l2) in
  aux l [];;

let rec map2 f l1 l2 = match l1, l2 with
  [],[] -> []
  | h::t,[] -> raise (Invalid_argument "map2")
  | [],h::t -> raise (Invalid_argument "map2")
  | h1::t1,h2::t2 -> (f h1 h2)::map2 f t1 t2;;

(*funcion para prueba de map2
let f2 x y = 2*x + y;;*)

let rec fold_left f init = function
  [] -> init
  | h::t -> fold_left f (f init h) t;;

  let rec fold_right f l init = match l with
  [] -> init
  | h::t -> f h (fold_right f t init);;



PRÁCTICA 8 - mylist2.ml, mylist3.ml 

let hd = function
    h::_ -> h
  |[] -> raise(Failure "hd");;
 
let tl = function
    _::t -> t
  |[] -> raise(Failure "tl");;

let length l = 
  let rec aux i = function
    [] -> i
    |_::t -> aux (i+1) t
  in aux 0 l;;   

let rec compare_lengths l1 l2 = match l1,l2 with
  [],[] -> 0
 |[], _ -> -1
 | _,[] -> 1
 |_::t1, _::t2 -> compare_lengths t1 t2;;

 
let nth l i = 
  if i < 0 then raise(Invalid_argument "nth")
  else let rec nthAux l i = match l,i with
        |[], _ -> raise(Failure "nth")
        |h::_, 0 -> h
        |_::t, _ -> nthAux t (i-1) in
  nthAux l i;; 
  
let rec append l1 l2 = match l1 with
  [] -> l2
 |h::t -> h :: append t l2;;  

let rec find f l = match l with
  [] -> raise(Not_found)
 |h::t -> if f h then h else find f t;; 

let rec for_all f l = match l with
  [] -> true
 |h::t -> f h && for_all f t;;

let rec exists f l = match l with
 [] -> false
|h::t -> f h || exists f t;;

let rec mem i l = match l with
  [] -> false
  |h::t -> h = i || mem i t;;

let rev l = 
  let rec aux l1 = function
    [] -> l1
   |h::t -> aux (h::l1) t in
  aux [] l;;  

let filter f l = 
  let rec aux f l1 = function
    [] -> rev l1
   |h::t -> if f h then aux f (h::l1) t 
            else aux f l1 t in
  aux f [] l;;                  

let find_all f l = filter f l;;

let partition f l =
  let rec aux f l1 l2 = function
    [] -> (rev l1, rev l2)
   |h::t -> if f h then aux f (h::l1) l2 t
            else aux f l1 (h::l2) t in
  aux f [] [] l;;   

let rec split = function 
  (h1,h2)::[] -> ([h1],[h2])
 |[] -> ([],[])
 |(h1,h2)::t -> let fst, scd = split t in
                (h1::fst, h2::scd);;     
                
let rec combine l1 l2 = match l1,l2 with
  [],[] -> []
 |[], l2 -> raise(Invalid_argument "List.combine")
 |l1, [] -> raise(Invalid_argument "List.combine")
 |h1::t1, h2::t2 -> (h1,h2)::combine t1 t2;;

let init len f = 
  if len < 0 then raise(Invalid_argument "init")
  else let rec aux f l = function
      0 -> l
     |n -> aux f (f n:: l) (n-1) in
  aux f [] len;;    
 
let rec rev_append l1 l2 =match l1 with
  [] -> l2
 |h::t -> rev_append t (h::l2);;

 let rev_map f l =
  let rec aux f l1 l2 = match l1 with
    | [] -> l2
    | h::t -> aux f t ((f h)::l2)
  in aux f l [];;
  
let rec concat = function 
  [] -> []
 |[]::t -> concat t
 |(h1::t1)::t -> h1:: concat (t1::t);;  

let rec flatten = function 
  [] -> []
 |[]::t -> flatten t
 |(h1::t1)::t -> h1:: flatten (t1::t);; 

let map f l =
  let rec aux f l1 = function
    [] -> rev l1
   |h::t -> aux f (f h::l1) t in
  aux f [] l;;
  
let rec map2 f l1 l2 = match l1, l2 with
  [],[] -> []
 |[],_ -> raise(Invalid_argument "map2")
 |_,[] -> raise(Invalid_argument "map2")
 |h1::t1, h2::t2 -> (f h1 h2)::map2 f t1 t2;; 

let rec fold_left f a = function
  [] -> a
 |h::t -> fold_left f (f a h) t;;
 
let rec fold_right f l a = match l with
  [] -> a
 |h::[]-> f h a
 |h::t -> f h (fold_right f t a);;

--------------------------------------------------------------------------------------------

let rec remove a = function
  [] -> []
  | h::t -> if h = a then t
            else h::remove a t;;

let rec remove_all a = function
  [] -> []
  | h::t -> if h = a then remove_all a t
            else h::remove_all a t;;
    
let rec ldif l1 = function
  [] -> l1
  | h2::t2 -> ldif (remove_all h2 l1) t2;;

let rec lprod l1 l2 = match l1, l2 with
  | _,[] -> []
  | [],_ -> []
  | h1::t1,h2::t2 -> (h1,h2)::(lprod [h1] t2)@(lprod t1 l2);;

let rec divide = function
  [] -> ([],[])
  | h1::[] -> ([h1],[])
  | h1::h2::t -> let l1, l2 = divide t in
                (h1::l1,h2::l2);;



PRÁCTICA 9 - ej91.ml, fact.ml, msort.ml, qsort.ml 

let to0from n =
  let rec aux m l = 
   if m > n then l 
   else aux (m+1) (m::l) in
  aux 0 [];;  

let fromto m n =
  let rec aux l n = 
    if m > n then l
    else aux (n::l) (n-1)
  in aux [] n;;

let incseg l = 
  let rec aux acum = function
    [] -> List.rev acum
   |h::t -> aux ((List.fold_left (+) h acum)::acum) t 
  in aux [] l;;   

let remove x l = 
  let rec aux acum x = function
    []-> List.rev acum
   |h::t -> if h = x then List.rev_append acum t
            else aux (h::acum) x t
  in aux [] x l;;         

let compress l =
  let rec aux acum = function
    [] -> []
   |h1::[] -> List.rev (h1::acum)  
   |h1::h2::t -> if h1 = h2 then aux acum (h1::t) 
                 else aux (h1::acum) (h2::t)
  in aux [] l;; 

--------------------------------------------------------------------------------------

let rec fact = function
  0 -> 1
  | n -> n * fact (n - 1);;
try
  print_endline (string_of_int (fact (int_of_string Sys.argv.(1))))
with
  Failure _
  | Stack_overflow
  | Invalid_argument _ -> print_endline "fact: argumento inválido"

------------------------------------------------------------------------------------------

let rec divide l = match l with
    h1::h2::t -> let t1, t2 = divide t in (h1::t1, h2::t2)
  | _ -> l, [];;


let rec merge = function
    [], l | l, [] -> l
  | h1::t1, h2::t2 -> if h1 <= h2 then h1 :: merge (t1, h2::t2)
                      else h2 :: merge (h1::t1, t2);;


let rec msort1 l = match l with
    [] | _::[] -> l
  | _ -> let l1, l2 = divide l in
         merge (msort1 l1, msort1 l2);;

(* Si, al ser no terminales cuando una lista sobrepase cierto número de
elementos se producira un agoramiento de la pila. *)

let l2 = List.init 100000 (fun x -> Random.int 100000);;


(* divide y merge terminales. *)

let divide' l =
	let rec divide'' acum1 acum2 = function
      [] -> (acum1, acum2)
    | h1::[] -> (h1::acum1, acum2)
		| h1::h2::t -> divide'' (h1::acum1) (h2::acum2) t
  in divide'' [] [] l;;


let rec merge' ord (l1, l2) =
  let rec merge'' acum = function
		  [],l | l,[] -> List.rev_append acum l
    | h1::t1, h2::t2 -> if ord h1 h2 then merge'' (h1::acum) (t1, h2::t2)
  				              else merge'' (h2::acum) (h1::t1, t2)
  in merge'' [] (l1, l2);;


let rec msort2 ord l = match l with
    [] | _::[] -> l
  | _ -> let l1, l2 = divide' l in
             merge' ord ((msort2 ord l1), (msort2 ord l2));;

(*
   Se realizan mediciones para la oredención de un mismo vector de números
   aleatorios de 70000 elementos.
   let l3 = List.init 70000 (fun _ -> Random.int 70000);;
   Sys.time();;
   - : float = 0,942955
   msort1 (<=) l3;;
   Sys.time();;
   - : float = 1,091294
   qsort2 (<=) l3;;
   Sys.time();;
   - : float = 1,253578
   msort2 (<=) l3;;
   Sys.time();;
   - : float = 1,404197
   1,091294 - 0,942955 = 0,148339 msort1
   1,253578 - 1,091294 = 0,162284  qsort2
   1,404197 - 1,253578 = 0,150619 msort2
   Las tres funciones de ordenación tienen un rendimiento bastante similar.
*)

---------------------------------------------------------------------------------------

let rec qsort1 ord = function
    [] -> []
   |h::t -> let after, before = List.partition (ord h) t in
  qsort1 ord before @ h :: qsort1 ord after;;


let rec qsort2 ord =
  let append' l1 l2 = List.rev_append (List.rev l1) l2 in
  function
      [] -> []
    | h::t -> let after, before = List.partition (ord h) t in
              append' (qsort2 ord before) (h :: qsort2 ord after);;


(* qsort2 tiene la ventaja de que nos permite ordenar listas más grandes que
qsort1 ya que no se utiliza @, pero se tiene que tener en cuenta que igualmente
se puede producir agotamiento del stack ya que tampoco es terminal. *)

(* Ejemplo de lista que produce agotamiento del stack
 * con qsort1 y no con qsort2 *)
let l1 = List.init 500000 (fun _ -> Random.int 500000);;


(* qsort2 tiene la desventaja que es más lento que qsort1, esto se explica
debido a que en el append' que está definido dentro de qsort2 se está
recorriendo dos veces l1 (se le da la vuelta dos veces). *)

(*
  Se realizan mediciones para la oredención de un mismo vector de números
  aleatorios de 250000 elementos.
  let l2 = List.init 250000 (fun _ -> Random.int 250000);;
  Sys.time();;
  - : float = 162.515771
  qsort1 (<=) l2;;
  Sys.time();;
  - : float = 163.093449
  qsort2 (<=) l2;;
  Sys.time();;
  - : float = 163.778185
  163,093449 - 162,515771 = 0,577678  qsort1
  163,778185 - 163,093449 = 0,684736  qsort2
  1 - (0,577678 / 0,684736) = 0,156 -> qsort2 es aproximadamente un 15,6%
  más lenta que qsort1
*)



PRÁCTICA 10 - tour.ml, shortest.ml 

let salta s (i1,j1)(i2,j2)  =
  (i2 = i1 && abs(j2 -j1) <= s) ||
  (j2 = j1 && abs(i2 -i1) <= s);;

let next_pos p s l = 
  (List.filter(salta s p ) l) ;;


let remove x l = 
  let rec aux x l1 l2 = match l1 with 
    [] -> l2
    | h::t -> if h = x then aux x t l2 
              else aux x t (h::l2)
  in aux x (List.rev l) [];;
  

let tour m n l s = 
  let rec completa promesa p saltos arboles = 
    if (p = (m,n)) then List.rev promesa (*Si llegamos a la ultima posicion devolvemos solucion*)
    else match saltos with 
    [] -> raise (Not_found)
    | h::t -> let l = remove h arboles in try completa (h::promesa) h (next_pos h s l) l with
                                          Not_found -> completa promesa p t l
  in if (List.mem (1,1) l) then let l = remove (1,1) l in completa [(1,1)] (1,1) (next_pos (1,1) s l) l
    else raise (Not_found);;

-------------------------------------------------------------------------------------------

let a_salto d (x1,y1) (x2,y2) = 
  (x1 = x2 && y1 <> y2 && abs(y1-y2)<= d) 
  || (y1 = y2 && x1 <> x2 &&abs(x1-x2) <= d);;


   let posible_salto camino d trees = 
    List.filter (fun a -> a_salto d (List.hd camino) a && not(List.mem a camino)) trees



let (@) l1 l2 = 
    List.rev_append (List.rev l1) l2 



let shortest_tour m n trees d = 
  let rec saltos caminos = match caminos with
        [] -> raise Not_found
        |h::t -> if List.hd h = (m,n) then List.rev h
                 else match List.map (fun s -> s::h)(posible_salto h d trees) with
                    [] -> saltos t
                    |s -> saltos (t @ s)
  in saltos [[(1,1)]];;



PRÁCTICA 11 - bin_tree.ml, breadth_first.ml, g_tree.ml, tsort.ml

type 'a bin_tree =
    Empty
  | Node of 'a * 'a bin_tree * 'a bin_tree
;;

let rec map_tree f = function
    Empty -> Empty
  | Node (x, l, r) -> Node (f x, map_tree f l, map_tree f r);;

let rec fold_tree f a = function
    Empty -> a
  | Node (x, l, r) -> f x (fold_tree f a l) (fold_tree f a r)
;;

let rec sum t = 
  let aux x l r = 
    x + l + r 
  in fold_tree aux 0 t;;

let rec prod t = 
  let aux x l r = 
    x *. l *. r 
  in fold_tree aux 1. t;;

let rec size t =  
  let aux x l r = 
    1 + l + r
  in fold_tree aux 0 t;;

let rec height t =   
  let aux x l r = 
    1 + max l r
  in fold_tree aux 0 t;;

let rec inorder t = 
  let aux x l r = 
    l @ [x] @ r
  in fold_tree aux [] t;;

let rec mirror t = 
  let aux x l r = 
    Node (x, r, l)
  in fold_tree aux Empty t;;

------------------------------------------------------------------------------------------

open G_tree;;

let rec breadth_first = function
    Gt (x, []) -> [x]
  | Gt (x, (Gt (y, t2))::t1) -> x :: breadth_first (Gt (y, t1@t2));;

let breadth_first_t gt = 
  let rec aux gt l = match gt with 
    Gt (x, []) -> List.rev (x::l)
  | Gt (x, (Gt (y, t2))::t1) -> aux (Gt(y, List.rev_append (List.rev t1) t2)) (x::l)
  in aux gt [];;

let dtree n = 
  let rec aux t n = 
    if n > 0 then
      aux (Gt (n,[t])) (n-1)
    else t
  in aux (Gt (0,[])) n;;

let t2 = dtree 200000;;
  
  (*Para breadth_first se obtiene: Stack overflow during evaluation (looping recursion?).
     La versión terminal hace el cálculo.*)

------------------------------------------------------------------------------------------

type 'a g_tree =
  Gt of 'a * 'a g_tree list
;;

let rec size = function 
    Gt (_, []) -> 1
  | Gt (r, h::t) -> size h + size (Gt (r, t))
;;

let rec height = function
    Gt (_, []) -> 1
  | Gt (_, l) -> 1 +
    let rec max n = function 
      [] -> n
    | h::t -> let ah = height h in if ah > n then max ah t
                                    else max n t
    in max 0 l;;     
  
let rec leaves = function
    Gt (r,[]) -> r::[]
  | Gt (_, l) -> List.concat (List.map leaves l);;

let rec mirror = function 
    Gt (r,[]) -> Gt (r, [])
  | Gt (r, l) -> Gt (r, List.map mirror (List.rev l));;

let rec preorder = function 
  Gt(r,l) -> r :: List.concat (List.map preorder l);;

let rec postorder = function
  Gt(r,l) -> List.concat (List.map postorder l) @ [r];;

------------------------------------------------------------------------------------------

open Bin_tree;;

let rec insert_tree ord x t= match t with
 Empty -> Node (x,Empty,Empty)
 | Node (a,l,r) -> 
  if (ord x a)
    then Node(a, insert_tree ord x l, r)
  else if (ord a x)
    then Node(a, l, insert_tree ord x r)
  else t;;

let tsort ord l =
  inorder (List.fold_left (fun a x -> insert_tree ord x a) Empty l);;



PRÁCTICA 12 

context.ml----------------------------------------------------------------------------------

type 'a context =
  (string * 'a) list;;

exception No_binding of string;;

let empty_context = [];;

let get_binding ctx name =
  try List.assoc name ctx with
  Not_found -> raise (No_binding name);;

let rec add_binding ctx name v = 
  (name, v)::ctx ;;

lib.ml--------------------------------------------------------------------------------------

open Context

exception Function_not_defined of string;;

let funs =  
  [("sqrt", sqrt); ("exp", exp);
  ("ln", log); ("round", Float.round)];;

let ctx = 
  let rec aux context left = match left with
    [] -> context
    | (h1,h2)::t -> aux (add_binding context h1 h2) t
  in aux empty_context funs;;

let get_function s =
  try get_binding ctx s with No_binding _ -> raise (Function_not_defined s);;

arith.ml------------------------------------------------------------------------------------

open Context;;
open Lib;;

type arith_oper =
    Opp;;

type arith_bi_oper =
    Sum | Sub | Prod | Div | Mod | Pow;;

type arith =
    Float of float
  | Var of string
  | Arith_op of arith_oper * arith
  | Arith_bi_op of arith_bi_oper * arith * arith
  | Fun_call of string * arith;;

let rec eval ctx = function
    Float f ->
      f

  | Var name ->
      get_binding ctx name 

  | Arith_op (oper, ar) -> (
    let a = eval ctx ar in match oper with
    Opp -> (-1.) *. a)

  | Arith_bi_op (oper, arith1, arith2) -> (let 
    a1, a2 = eval ctx arith1, eval ctx arith2 in 
    match oper with 
      Sum ->  a1 +. a2
    | Sub ->  a1 -. a2
    | Prod -> a1 *. a2
    | Div ->  a1 /. a2
    | Mod ->  float_of_int((int_of_float a1) mod (int_of_float a2))
    | Pow ->  Float.pow a1 a2)

  | Fun_call (s, ar) -> 
    get_function s (eval ctx ar)

;;

command.ml---------------------------------------------------------------------------------

open Context;;
open Arith;;

exception End_of_program;;

type command =
    Eval of arith
  | Var_def of string * arith
  | Quit;;

let rec run ctx = function
    Eval e ->
      let f = eval ctx e in
      let _ = print_endline (string_of_float f) in
      ctx
  | Var_def (s, ar) -> 
    let f = eval ctx ar in
    let _ = print_endline (s ^ " = " ^ string_of_float f) in
    add_binding ctx s f 
  | Quit -> raise (End_of_program) ;;

main.ml-------------------------------------------------------------------------------------

open Parsing;;
open Lexing;;

open Context;;
open Lib;;
open Arith;;
open Command;;
open Parser;;
open Lexer;;

let rec loop ctx =
  print_string ">> ";


  try 
    let ctx = run ctx  (s token (from_string (read_line ()))) in loop ctx 
  with 
   End_of_program -> ctx
  | No_binding s -> let _ = print_endline ("Var "^s^" not defined") in loop ctx
  | Function_not_defined s ->  let _ = print_endline ("Function "^s^" not defined") in loop ctx
  | Parse_error -> let _ = print_endline "Syntax error" in loop ctx
  | Lexical_error -> let _ = print_endline "Lexical error" in loop ctx
;;

let _ = print_endline "Floating point calculator..." in
let _ = loop empty_context in
print_endline "... bye!!!";;
