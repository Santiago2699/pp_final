

(*********************************************************
 * EJEMPLO DE DEFINICIÓN DE CLASE (class point2D) 
 *********************************************************)

(*Cabecera del objeto: nombre + atributos (no necesario añadir el tipo)*)
 # class point2D (x_init, y_init) (* coords. del punto *)  =	
 (*Cuerpo de la clase*)
 object (self) (*ALIAS: entre paréntesis id de alias para que una instancia
    pueda referenciarse a si misma (.this en java)*)
  (*ATRIBUTOS, simpre privados en ocaml*)
   (* coordenadas XX' e YY' *)
   val mutable x = x_init (*se le da el valor inicial OBLIGATORIO*)
   val mutable y = y_init  (*ultilizamos mutable para que sean variables,
                            sino serian static*)
  (*MÉTODOS, por defencto públicos, para que sean privados ponemos private después de method*)
   (* getters acceso a coord. *)
   method get_x = x
   method get_y = y
 
   (* setters asignacion coord. *)
   (*<- operador de asignación para métodos*)
   method set_x x' = x<-x'
   method set_y y' = y<-y'
   
   (* reasignar coordenadas de forma absoluta o relativa *)
   method moveto (x',y') = x<-x'; y<-y'
   method rmoveto (dx,dy) = self#moveto(x+dx, y+dy) (*llamada a método con # en vez de . como java
                                                      obligatorio usar siempre alias*)
 
   (* toString() *)
   method to_string () = "( "^(string_of_int x)^", "^(string_of_int y)^" )"
 
 end;;
 
 (*SALIDA DEL COMPILADOR INTERACTIVO*)
 (*muestra en orden alfabético*)
 class point2D : 
   int * int ->
   object
     val mutable x : int
     val mutable y : int
     method get_x : int
     method get_y : int
     method moveto : int * int -> unit
     method rmoveto : int * int -> unit
     method set_x : int -> unit
     method set_y : int -> unit
     method to_string : unit -> string
   end
  
 
 (*-----------------------------------------------------
  * EJEMPLO DE CREACIÓN DE INSTANCIA
  *-----------------------------------------------------*)
 (*en Ocaml no hay constructor de clase, se utiliza la palabra reservada
    new seguida del nombre de la clase con los parámetros necesarios.*)

 # let p1 = new point2D (1,11);;  
 val p1 : point2D = <obj>

 # let p2 = new point2D (2,22);;  
 val p2 : point2D = <obj>
 
 
 (*-----------------------------------------------------
  * EJEMPLOS DE LLAMADA A MÉTODOS 
  *-----------------------------------------------------*)
 
 # p1#get_x;;
 - : int = 1

 # p1#get_y;;
 - : int = 11

 # p1#to_string ();;
 - : string = "( 1, 11 )"
 
 # p1#set_x 10;;
 - : unit = ()

 # p1#set_y 20;;
 - : unit = ()

 # p1#to_string ();;
 - : string = "( 10, 20 )"
 
 
 (*-----------------------------------------------------
  * EJEMPLOS DE MANEJO DE INSTANCIAS COMO OTRA DEFINCIÓN MÁS 
  *-----------------------------------------------------*)
 
 # let listilla = [("uno", p1); ("dos", p2) ];; 
 val listilla : (string * point2D) list = [("uno", <obj>); ("dos", <obj>)]
 
 # (snd (List.hd listilla));;
 - : point2D = <obj>

 # (snd (List.hd listilla))#to_string();;
 - : string = "( 10, 20 )"
 
 # List.assoc "dos" listilla;;
 - : point2D = <obj>

 # (List.assoc "dos" listilla)#to_string();;
 - : string = "( 2, 22 )"
 
 # p2#moveto (3,33);;
 - : unit = ()

 # p2#to_string ();;
 - : string = "( 3, 33 )"
 
 # (List.assoc "dos" listilla)#to_string();;
 - : string = "( 3, 33 )"
 
 
 # let fcreate_point2D xx yy = new point2D (xx, yy);; 
 val fcreate_point2D : int -> int -> point2D = <fun>

 # let p4 = fcreate_point2D 4 44;;
 val p4 : point2D = <obj>

 # p4#to_string ();;
 - : string = "( 4, 44 )"
 
 
 
 (*-----------------------------------------------------
  * IMMEDIATE OBJECTS: (ocaml nos permite crear clases sin haber
  definido previamente el cuerpo de la clase)
  *-----------------------------------------------------*)
 
 (* --- CLASE SENCILLA DE EJEMPLO (class point1D)  --------------------*)
  
  # class point1D x_init =
       object
         val mutable x = x_init
         method get_x = x
         method rmoveto d = x <- x + d
       end;;
 
  (*SALIDA EN COMPILADOR INTERACTIVO*)
 class point1D :
   int ->
   object
     val mutable x : int
     method get_x : int
     method rmoveto : int -> unit
   end
   
  (*El tipado de objetos en ocaml es estructural: depende del conjunto de firmas de sus MÉTODOS PÚBLICOS.
     En este caso el tipo quetendrían las instancias de la clase point1D sería < get_x: int; rmoveto: int -> unit >*)

 # let p1d = new point1D 0;;
 val p1d : point1D = <obj>

 (*a p1d se le asigna el tipo de instancia: < get_x: int; rmoveto: int -> unit > pero por abreviatura se le asigna
    el nombre point1D para hacer referencia a todo el churro anterior.*)
 
 # p1d#get_x;;
 - : int = 0
 # p1d#rmoveto 3;;
 - : unit = ()
 # p1d#get_x;;
 - : int = 3
 
 
 (* --- EJEMPLO DE IMMEDIATE OBJECT "equivalente" --------------------*)
   
 # let o1d =
   object
     val mutable x = 0
     method get_x = x
     method rmoveto d' = x <- x + d'
   end;;
   
  (*SALIDA COMPILADOR INTERACTIVO*)
 val o1d : < get_x : int; rmoveto : int -> unit > = <obj>
 
 
 # o1d#get_x;;
 - : int = 0
 # o1d#rmoveto 3;;
 - : unit = ()
 # o1d#get_x;;
 - : int = 3
 
 
 (* --- EJEMPLO DE FACTORY FUNCTION --------------------*)
 (*Función que dado una coordenada nos devuelve un objeto.
    "crear un constructor de un objeto inmediato"*)
 
 # let factoria_pinmediato1D (xinit:int) = 
   object
     val mutable x = xinit
     method get_x = x
     method rmoveto d' = x <- x + d'
   end;;
   
 val factoria_pinmediato1D : int -> < get_x : int; rmoveto : int -> unit > = <fun>
 
 
 # let i5 = factoria_pinmediato1D 5;;
 val i5 : < get_x : int; rmoveto : int -> unit > = <obj>
 # let i6 = factoria_pinmediato1D 6;;
 val i6 : < get_x : int; rmoveto : int -> unit > = <obj>
 # abs (i5#get_x - i6#get_x);; (*medir distancia entre dos puntos*)
 - : int = 1
 
 
 
 (*********************************************************
  * AGREGACIÓN 
  *********************************************************)
 
 # class edge2D (a: point2D) (b: point2D) =
   object
       val vertexes = (a,b)
       method get_vertexes = vertexes     
   end;;
 
 class edge2D :
   point2D ->
   point2D ->
   object
     val vertexes : point2D * point2D
     method get_vertexes : point2D * point2D
   end
 
 
 (*********************************************************
  * HERENCIA SIMPLE:
  *     Ejemplo: añadir método equals a point2D en base a coordenadas
  *********************************************************)
 
  (*Definición de una subclase con superclase: punto que además
     contiene un método equals.*)
 # class point2Deq (x_init, y_init) (* coords. del punto *) =	
 object (self:'self) (*obligar a que el tipo pasado a los métodos
    sea el de la subclase*)
     
     inherit point2D (x_init, y_init) (*la calse que estamos definiendo es subclase de point2D
                                        usamos nombre superclase + atributos + (as "alias")*)
     
   (* NEW: equals() *)
   (*Debemos mantener el tipo de los métodos al reescribirlos*)
   method equals (o:'self) = (self#get_x = o#get_x) && (self#get_y = o#get_y)
   
 end;;
 
 (*SALIDA DEL COMPILADOR INTERACTIVO*)
 class point2Deq :
   int * int ->
   object ('a) ()
     val mutable x : int
     val mutable y : int
     method equals : 'a -> bool (*nuevo método equals, donde 'a es el tipo de point2Deq, no polimórfico*)
     method get_x : int
     method get_y : int
     method moveto : int * int -> unit
     method rmoveto : int * int -> unit
     method set_x : int -> unit
     method set_y : int -> unit
     method to_string : unit -> string
   end
 
 # let peq1 = new point2Deq (1,2);;
 val peq1 : point2Deq = <obj>
 # let peq2 = new point2Deq (1,2);;
 val peq2 : point2Deq = <obj>
 # let peq3 = new point2Deq (3,3);;
 val peq3 : point2Deq = <obj>
 
 # peq1#equals peq2;;
 - : bool = true
 # peq1#equals peq3;;
 - : bool = false
  
  
   
 (*********************************************************
  * IGUALDAD DE OBJETOS 
  *********************************************************)
 
  
 (*-----------------------------------------------------
  *  IGUALDAD FÍSICA: si estamos o no ante la misma instancia
  *-----------------------------------------------------*)
 
 # let pp1 = new point2D (0,0);;
 val pp1 : point2D = <obj>
 # let pp2 = new point2D (0,0);;
 val pp2 : point2D = <obj>
 # let pp3 = new point2D (3,33);;
 val pp3 : point2D = <obj>
 
 
 # pp1=pp2;;
 - : bool = false
 
 # let pp1'=pp1;;
 val pp1' : point2D = <obj>
 # pp1'=pp1;;
 - : bool = true
 
 
        
 (*-----------------------------------------------------
  * IGUALDAD RESPECTO A CRITERIO (1): FUNCIÓN
  * ej. "dos puntos son iguales si tienen mismas coordenadas"
  *-----------------------------------------------------*)
 
(*creando una nueva función que se pasan dos objetos
  (caso de abajo) ...   *)
 # let are_equal_points (a: point2D) (b: point2D) = 
       (a#get_x = b#get_x) && (a#get_y = b#get_y);;
 val are_equal_points : point2D -> point2D -> bool = <fun>
 
 # are_equal_points pp1 pp2;;
 - : bool = true
 # are_equal_points pp1 pp3;;
 - : bool = false
 
 
 (*-----------------------------------------------------
  * IGUALDAD RESPECTO A CRITERIO (2): MÉTODO PROPIO
  *     Ejemplo: añadir método equals a point2D en base a coordenadas
  *********************************************************)
 
  (* ...  o implementar un método equals en la clase como en el caso de 
   point2Deq*)
 # class point2Deq (x_init, y_init) (* coords. del punto *) =	
 object (self:'self)
     
     inherit point2D (x_init, y_init)
     
   (* NEW: equals() *)
   method equals (o:'self) = (self#get_x = o#get_x) && (self#get_y = o#get_y)
   
 end;;
 
 (*SALIDA DEL COMPILADOR INTERACTIVO*)
 class point2Deq :
   int * int ->
   object ('a)
     val mutable x : int
     val mutable y : int
     method equals : 'a -> bool
     method get_x : int
     method get_y : int
     method moveto : int * int -> unit
     method rmoveto : int * int -> unit
     method set_x : int -> unit
     method set_y : int -> unit
     method to_string : unit -> string
   end
 
 # let peq1 = new point2Deq (1,2);;
 val peq1 : point2Deq = <obj>
 # let peq2 = new point2Deq (1,2);;
 val peq2 : point2Deq = <obj>
 # let peq3 = new point2Deq (3,3);;
 val peq3 : point2Deq = <obj>
 
 # peq1#equals peq2;;
 - : bool = true
 # peq1#equals peq3;;
 - : bool = false
  
  
 
 